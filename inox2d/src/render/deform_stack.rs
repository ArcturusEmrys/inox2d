use std::collections::HashMap;
use std::mem::swap;

use glam::Vec2;

use crate::math::deform::{deform_by_parent_triangle, linear_combine, vector_decompose_matrix, Deform};
use crate::node::components::{DeformSource, DeformStack, Mesh, TransformStore};
use crate::puppet::{InoxNodeTree, World};

impl DeformStack {
	pub(crate) fn new(deform_len: usize) -> Self {
		Self {
			deform_len,
			stack: HashMap::new(),
		}
	}

	/// Reset the stack. Ready to receive deformations for one frame.
	pub(crate) fn reset(&mut self) {
		for enabled_deform in self.stack.values_mut() {
			enabled_deform.0 = false;
		}
	}

	/// Combine the deformations received so far according to some rules, and write to the result
	pub(crate) fn combine(&self, nodes: &InoxNodeTree, node_comps: &World, result: &mut [Vec2]) {
		if result.len() != self.deform_len {
			panic!("Required output deform dimensions different from what DeformStack is initialized with.")
		}

		// Single pass might require more structure changes. I can't figure it out how without.
		// I will try to make it work as straightforward as possible.

		// Temporary. Nested meshgroup for todo
		let mut maybe_meshgroup_uuid = None;
		let mut maybe_meshgroup_deform = None;
		let mut maybe_node_id = None;

		let direct_deforms = self.stack.iter().filter_map(|(deform_source, enabled_deform)| {
			if enabled_deform.0 {
				match (&deform_source, &enabled_deform.1) {
					(DeformSource::Param(_), Deform::Direct(ref direct_deform)) => Some(direct_deform),
					(DeformSource::MeshGroup(_, mg_uuid), Deform::FromMeshGroup(ref mg_deform, ref node_uuid)) => {
						maybe_meshgroup_uuid = Some(mg_uuid);
						maybe_meshgroup_deform = Some(mg_deform);
						maybe_node_id = Some(node_uuid);
						None
					}
					_ => todo!(), // panic? It's illegal
				}
			} else {
				None
			}
		});
		linear_combine(direct_deforms, result);

		if let (Some(node_id), Some(meshgroup_uuid), Some(meshgroup_deform)) =
			(maybe_node_id, maybe_meshgroup_uuid, maybe_meshgroup_deform)
		{
			let _child_node = nodes.get_node(*node_id).unwrap();
			let child_mesh = node_comps.get::<Mesh>(*node_id).unwrap();
			let child_init_verts = &child_mesh.vertices; // must have a mesh if already on deform stack
			let child_direct_verts = child_init_verts
				.iter()
				.zip(result.iter())
				.map(|(point, deform)| point + deform);

			let child_transform = node_comps.get::<TransformStore>(*node_id).unwrap().absolute;
			// Need this because triangle test is in meshgroup's space
			let meshgroup_transform = node_comps.get::<TransformStore>(*meshgroup_uuid).unwrap().absolute;
			let to_meshgroup_space = meshgroup_transform.inverse() * child_transform;

			// take account of child deform results
			// the "result" so far should be applied on the initial mesh, not the transformed mesh
			let child_meshgroup_verts: &Vec<Vec2> = &child_direct_verts
				.map(|point| to_meshgroup_space.transform_point3(point.extend(0.0)).truncate())
				.collect();

			// child_verts appears to be incorrect
			// it should include its own deform

			let meshgroup_mesh = node_comps.get::<Mesh>(*meshgroup_uuid).unwrap();
			// 	// TODO: bitmask optimization
			let triangle_by_point: Vec<Option<u16>> = meshgroup_mesh.test(child_meshgroup_verts.iter()).collect();

			let mut deform_results = vec![Vec2::default(); child_meshgroup_verts.len()];
			let mut grouped_points: HashMap<u16, (Vec<usize>, Vec<&Vec2>)> = HashMap::new();

			for ((idx, point), tri_idx) in child_meshgroup_verts
				.iter()
				.enumerate()
				.zip(triangle_by_point.into_iter())
			{
				if let Some(id) = tri_idx {
					let (indices, pts) = grouped_points.entry(id).or_default();
					indices.push(idx);
					pts.push(point);
				}
			}

			for (tri_idx, (indices, points_in_tri)) in grouped_points {
				let tri = meshgroup_mesh.get_triangle(tri_idx);
				let decompose_matrix = vector_decompose_matrix(tri[1] - tri[0], tri[2] - tri[0]);
				let parent_deforms = [
					meshgroup_deform[meshgroup_mesh.indices[3 * tri_idx as usize] as usize],
					meshgroup_deform[meshgroup_mesh.indices[(3 * tri_idx + 1) as usize] as usize],
					meshgroup_deform[meshgroup_mesh.indices[(3 * tri_idx + 2) as usize] as usize],
				];

				let deform_by_triangle =
					deform_by_parent_triangle(&decompose_matrix, tri[0], &parent_deforms, points_in_tri.into_iter());

				for (idx, deform_point) in indices.into_iter().zip(deform_by_triangle) {
					deform_results[idx] = deform_point;
				}
			}
			// linear_combine(vec![deform_results].iter(), result);

			// The calculated deform is in meshgroup's space,
			// so convert the deform back to the target mesh's space
			let to_node_space = child_transform.inverse() * meshgroup_transform;
			let deform_node_space = deform_results
				.iter()
				.map(|deform| to_node_space.transform_vector3(deform.extend(0.0)).truncate());

			result
				.iter_mut()
				.zip(deform_node_space)
				.for_each(|(sum, addition)| *sum += addition);
		}
	}

	/// Submit a deform from a source for a node.
	pub(crate) fn push(&mut self, src: DeformSource, mut deform: Deform) {
		match deform {
			Deform::Direct(ref direct_deform) => {
				if direct_deform.len() != self.deform_len {
					panic!("A direct deform with non-matching dimensions is submitted to a node.");
				}

				self.stack
					.entry(src)
					.and_modify(|enabled_deform| {
						if enabled_deform.0 {
							panic!("A same source submitted deform twice for a same node within one frame.")
						}
						enabled_deform.0 = true;

						swap(&mut enabled_deform.1, &mut deform);
					})
					.or_insert((true, deform));
			}
			// TODO: I don't know if we can add necessary information so we can use them during combine
			Deform::FromMeshGroup(_, _) => {
				self.stack
					.entry(src)
					.and_modify(|enabled_deform| {
						if enabled_deform.0 {
							panic!("A same source submitted deform twice for a same node within one frame.")
						}
						enabled_deform.0 = true;

						swap(&mut enabled_deform.1, &mut deform);
					})
					.or_insert((true, deform));
			}
		}
	}
}
